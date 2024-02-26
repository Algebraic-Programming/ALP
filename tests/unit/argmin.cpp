
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
	grb::Vector< size_t > index( n );
	grb::Vector< double > value( n );
	grb::Vector< std::pair< size_t, double > > left( n ), right( n ), out( n );

	rc = grb::set( value, 1.5 );
	if( rc == SUCCESS ) {
		rc = grb::set< grb::descriptors::use_index >( index, 0 );
	}
	if( rc == SUCCESS ) {
		rc = grb::zip( left, index, value );
	}
	if( rc == SUCCESS ) {
		rc = grb::eWiseLambda(
			[ &index, &n ]( const size_t i ) {
				index[ i ] = n - i;
			},
			index );
	}
	if( rc == SUCCESS ) {
		rc = grb::set( value, 3.5 );
	}
	if( rc == SUCCESS ) {
		rc = grb::setElement( value, 0.5, n / 2 );
	}
	if( rc == SUCCESS ) {
		rc = grb::zip( right, index, value );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t initialisation FAILED\n";
		return;
	}

	grb::operators::argmin< size_t, double > argminOp;
	grb::Monoid<
		grb::operators::argmin< size_t, double >,
		grb::identities::infinity
	> argminMonoid;

	// test 1
	rc = grb::eWiseApply( out, left, right, argminOp );
	if( rc != SUCCESS ) {
		std::cerr << "\t element-wise application of argmin FAILED\n";
		return;
	}
	if( grb::nnz( out ) != n ) {
		std::cerr << "\t element-wise argmin results in " << grb::nnz( out )
			<< " nonzeroes, but expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto & pair : out ) {
		if( pair.first == n / 2 ) {
			if( pair.second.second != 0.5 ) {
				std::cerr << "\t element-wise argmin results in unexpected entry ( "
					<< pair.first << ", [ " << pair.second.first << ", "
					<< pair.second.second << " ] ): expected value 0.5.\n";
				rc = FAILED;
			}
		} else if( pair.second.second != 1.5 ) {
			std::cerr << "\t element-wise argmin results in unexpected entry ( "
				<< pair.first << ", [ " << pair.second.first << ", "
				<< pair.second.second << " ] ): expected value 1.5.\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test 2
	std::pair< size_t, double > reduced;
	reduced.first = std::numeric_limits< size_t >::max();
	reduced.second = std::numeric_limits< double >::max();
	rc = grb::foldl( reduced, right, argminMonoid );
	if( rc != SUCCESS ) {
		std::cerr << "\t reduction via argmin (left-one) FAILED\n";
		return;
	}
	if( reduced.first != n / 2 || reduced.second != 0.5 ) {
		std::cerr << "\t reduction via argmin (left-one) has unexpected result ( "
			<< reduced.first << ", " << reduced.second << " ): expected ( "
			<< ( n / 2 ) << ", 0.5 ).\n";
		rc = FAILED;
		return;
	}

	// test 3
	reduced.first = std::numeric_limits< size_t >::max();
	reduced.second = std::numeric_limits< double >::max();
	rc = grb::foldr( left, reduced, argminMonoid );
	if( rc != SUCCESS ) {
		std::cerr << "\t reduction via argmin (right-any) FAILED\n";
		return;
	}
	if( reduced.second != 1.5 ) {
		std::cerr << "\t reduction via argmin (right-any) has unexpected result ( "
			<< reduced.first << ", " << reduced.second << " ): expected value 1.5.\n";
		rc = FAILED;
		return;
	}

	// test 4
	reduced.first = std::numeric_limits< size_t >::max();
	reduced.second = std::numeric_limits< double >::max();
	rc = grb::setElement( left, std::make_pair< size_t, double >( n / 2, 7.5 ), n / 2 );
	if( rc == SUCCESS ) {
		rc = grb::foldr( left, reduced, argminMonoid );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t reduction via argmin (right-any-except) FAILED\n";
		return;
	}
	if( reduced.second == n / 2 || reduced.second != 1.5 ) {
		std::cerr << "\t reduction via argmin (right-any-except) "
			<< "has unexpected result ( " << reduced.first << ", "
			<< reduced.second << " ): expected ( i, 1.5 ) with i "
			<< "not equal to " << ( n / 2 ) << "\n";
		rc = FAILED;
		return;
	}

	// test 5
	std::pair< int, float > sevenPi = { 7, 3.1415926535 };
	std::pair< int, float > minusOneTwo = { -1, 2 };
	std::pair< int, float > test;
	grb::operators::argmin< int, float > intFloatArgmin;
	rc = apply( test, sevenPi, minusOneTwo, intFloatArgmin );
	if( rc != SUCCESS ) {
		std::cerr << "\t application of argmin to scalars (I) FAILED\n";
		rc = FAILED;
	} else {
		if( test.first != -1 || test.second != 2 ) {
			std::cerr << "\t argmin to scalars (I) returns " << test.first << ", "
				<< test.second << " instead of -1, 2\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test 6
	test = { 10, 10.0 };
	rc = apply( test, minusOneTwo, sevenPi, intFloatArgmin );
	if( rc != SUCCESS ) {
		std::cerr << "\t application of argmin to scalars (II) FAILED\n";
		rc = FAILED;
	} else {
		if( test.first != -1 || test.second != 2 ) {
			std::cerr << "\t argmin to scalars (II) returns " << test.first << ", "
				<< test.second << " instead of -1, 2\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test 7
	test = sevenPi;
	rc = foldl( test, minusOneTwo, intFloatArgmin );
	if( rc != SUCCESS ) {
		std::cerr << "\t foldl of scalars (I) FAILED\n";
		rc = FAILED;
	} else {
		if( test.first != -1 || test.second != 2 ) {
			std::cerr << "\t foldl of scalars (I) returns " << test.first << ", "
				<< test.second << " instead of -1, 2\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test 8
	test = sevenPi;
	rc = foldr( minusOneTwo, test, intFloatArgmin );
	if( rc != SUCCESS ) {
		std::cerr << "\t foldr of scalars (I) FAILED\n";
		rc = FAILED;
	} else {
		if( test.first != -1 || test.second != 2 ) {
			std::cerr << "\t foldr of scalars (I) returns " << test.first << ", "
				<< test.second << " instead of -1, 2\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test 9
	test = minusOneTwo;
	rc = foldl( test, sevenPi, intFloatArgmin );
	if( rc != SUCCESS ) {
		std::cerr << "\t foldl of scalars (II) FAILED\n";
		rc = FAILED;
	} else {
		if( test.first != -1 || test.second != 2 ) {
			std::cerr << "\t foldl of scalars (II) returns " << test.first << ", "
				<< test.second << " instead of -1, 2\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test 10
	test = minusOneTwo;
	rc = foldr( sevenPi, test, intFloatArgmin );
	if( rc != SUCCESS ) {
		std::cerr << "\t foldr of scalars (II) FAILED\n";
		rc = FAILED;
	} else {
		if( test.first != -1 || test.second != 2 ) {
			std::cerr << "\t foldr of scalars (II) returns " << test.first << ", "
				<< test.second << " instead of -1, 2\n";
			rc = FAILED;
		}
	}

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
