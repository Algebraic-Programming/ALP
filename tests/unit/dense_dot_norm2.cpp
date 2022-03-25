
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

#include <alp.hpp>

using namespace alp;

void alp_program( const size_t &n, alp::RC &rc ) {

	// repeatedly used containers
	alp::Vector< double > left( n );
	alp::Vector< double > right( n );

	// test 1, init
	alp::Semiring<
		alp::operators::add< double >, alp::operators::mul< double >,
		alp::identities::zero, alp::identities::one
	> ring;
	// rc = alp::set( left, 1.5 ); // left = 1.5 everywhere
	// rc = rc ? rc : alp::set( right, -1.0 );
	// if( rc != SUCCESS ) {
	// 	std::cerr << "\t test 1 (dense, regular semiring): initialisation FAILED\n";
	// 	return;
	// }
	Scalar< double > out( 2.55 );

	// test 1, exec
	rc = alp::dot( out, left, right, ring );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 1 (dense, regular semiring): dot FAILED\n";
		return;
	}

	// // test 1, check
	// const double expected = 2.55 - static_cast< double >( n + n / 2 );
	// if( !utils::equals( out, expected, 2 * n + 1 ) ) {
	// 	std::cerr << "\t test 1 (dense, regular semiring): unexpected output "
	// 		<< "( " << out << ", expected "
	// 		<< ( 2.55 - static_cast< double >(n + n/2) )
	// 		<< " )\n";
	// 	rc = FAILED;
	// }
	// if( rc != SUCCESS ) {
	// 	return;
	// }

	// test 2, init
	// alp::Semiring<
	// 	alp::operators::add< double >, alp::operators::left_assign_if< double, bool, double >,
	// 	alp::identities::zero, alp::identities::logical_true
	// > pattern_sum_if;
	// rc = alp::clear( left );
	// rc = rc ? rc : alp::clear( right );
	// for( size_t i = 0; 2 * i < n; ++i ) {
	// 	rc = rc ? rc : alp::setElement( left, 2.0, 2 * i );
	// 	rc = rc ? rc : alp::setElement( right, 1.0, 2 * i );
	// }
	// if( rc != SUCCESS ) {
	// 	std::cerr << "\t test 2 (sparse, non-standard semiring) initialisation FAILED\n";
	// 	return;
	// }
	// out = 0;

	// // test 2, exec
	// rc = alp::dot( out, left, right, pattern_sum_if );
	// if( rc != SUCCESS ) {
	// 	std::cerr << "\t test 2 (sparse, non-standard semiring) dot FAILED\n";
	// 	return;
	// }

	// // test 2, check
	// if( !utils::equals( out, static_cast< double >( n ), 2 * n ) ) {
	// 	std::cerr << "\t test 2 (sparse, non-standard semiring), "
	// 		<< "unexpected output: " << out << ", expected " << n
	// 		<< ".\n";
	// 	rc = FAILED;
	// 	return;
	// }

	// // test 3, init
	// alp::Semiring<
	// 	alp::operators::add< int >, alp::operators::mul< int >,
	// 	alp::identities::zero, alp::identities::one
	// > intRing;
	// alp::Vector< int > x( n ), y( n );
	// rc = alp::set( x, 1 );
	// rc = rc ? rc : alp::set( y, 2 );
	// if( rc != alp::SUCCESS ) {
	// 	std::cerr << "\t test 3 (dense integer vectors) initialisation FAILED\n";
	// 	return;
	// }
	// int alpha = 0;

	// // test 3, exec
	// rc = alp::dot( alpha, x, y, intRing );
	// if( rc != alp::SUCCESS ) {
	// 	std::cerr << "\t test 3 (dense integer vectors) dot FAILED\n";
	// 	return;
	// }

	// // test 3, check
	// if( alpha != 2 * static_cast< int >(n) ) {
	// 	std::cerr << "\t test 3 (dense integer vectors) unexpected value "
	// 		<< alpha << ", expected 2 * n = " << (2*n) << ".\n";
	// 	rc = FAILED;
	// 	return;
	// }

	// // test 4, init
	// alp::Vector< int > empty_left( 0 ), empty_right( 0 );
	// // retain old value of alpha

	// // test 4, exec
	// rc = alp::dot( alpha, empty_left, empty_right, intRing );
	// if( rc != SUCCESS ) {
	// 	std::cerr << "\t test 4 (empty vectors) dot FAILED\n";
	// 	return;
	// }

	// // test 4, check
	// if( alpha != 2 * static_cast< int >(n) ) {
	// 	std::cerr << "\t test 4 (empty vectors) unexpected value "
	// 		<< alpha << ", expected 2 * n = " << (2*n) << ".\n";
	// 	rc = FAILED;
	// 	return;
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
