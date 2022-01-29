
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
	grb::Vector< size_t > index( n );
	grb::Vector< double > value( n );
	grb::Vector< std::pair< size_t, double > > left( n ), right( n ), out( n );

	rc = grb::set( value, 1.5 );
	if( rc == SUCCESS ) {
		rc = grb::set< grb::descriptors::use_index >( index, 0 );
	}
	if( rc == SUCCESS ) {
		rc = grb::zip( left, index, value ); // left = ( 0..n-1, 1.5 )
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
		rc = grb::zip( right, index,
			value ); // right = ( n-1..0, 3.5 ) except for ( n/2, 0.5 )
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t initialisation FAILED\n";
		return;
	}

	grb::operators::argmax< size_t, double > argmaxOp;
	grb::Monoid< grb::operators::argmax< size_t, double >, grb::identities::negative_infinity > argmaxMonoid;

	// test 1
	rc = grb::eWiseApply( out, left, right, argmaxOp );
	if( rc != SUCCESS ) {
		std::cerr << "\t element-wise application of argmax FAILED\n";
		return;
	}
	if( grb::nnz( out ) != n ) {
		std::cerr << "\t element-wise argmax results in " << grb::nnz( out ) << " nonzeroes, but expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto & pair : out ) {
		if( pair.second.first != n / 2 ) {
			if( pair.second.second != 3.5 ) {
				std::cerr << "\t element-wise argmax results in unexpected "
							 "entry ( "
						  << pair.first << ", [ " << pair.second.first << ", " << pair.second.second << " ] ): expected value 3.5.\n";
				rc = FAILED;
			}
		} else {
			if( pair.second.second != 1.5 ) {
				std::cerr << "\t element-wise argmax results in unexpected "
							 "entry ( "
						  << pair.first << ", [ " << pair.second.first << ", " << pair.second.second << " ] ): expected value 1.5.\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test 2
	std::pair< size_t, double > reduced;
	reduced.first = std::numeric_limits< size_t >::min();
	reduced.second = -std::numeric_limits< double >::infinity();
	rc = grb::foldl( reduced, right, argmaxMonoid );
	if( rc != SUCCESS ) {
		std::cerr << "\t reduction via argmax (left-one) FAILED\n";
		return;
	}
	if( reduced.first == n / 2 || reduced.second != 3.5 ) {
		std::cerr << "\t reduction via argmax (left-one) has unexpected result "
					 "( "
				  << reduced.first << ", " << reduced.second << " ): expected entry with index anything else than " << ( n / 2 ) << " and value 3.5.\n";
		rc = FAILED;
		return;
	}

	// test 3
	reduced.first = std::numeric_limits< size_t >::min();
	reduced.second = -std::numeric_limits< double >::infinity();
	rc = grb::foldr( left, reduced, argmaxMonoid );
	if( rc != SUCCESS ) {
		std::cerr << "\t reduction via argmax (right-any) FAILED\n";
		return;
	}
	if( reduced.second != 1.5 ) {
		std::cerr << "\t reduction via argmax (right-any) has unexpected "
					 "result ( "
				  << reduced.first << ", " << reduced.second << " ): expected value 1.5.\n";
		rc = FAILED;
		return;
	}

	// test 4
	reduced.first = std::numeric_limits< size_t >::min();
	reduced.second = -std::numeric_limits< double >::infinity();
	rc = grb::setElement( left, std::make_pair< size_t, double >( n / 2, 7.5 ), n / 2 );
	if( rc == SUCCESS ) {
		rc = grb::foldr( left, reduced, argmaxMonoid );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t reduction via argmax (right-one) FAILED\n";
		return;
	}
	if( reduced.first != n / 2 || reduced.second != 7.5 ) {
		std::cerr << "\t reduction via argmax (right-one) has unexpected "
					 "result ( "
				  << reduced.first << ", " << reduced.second << " ): expected ( " << ( n / 2 ) << ", 7.5 )\n";
		rc = FAILED;
		return;
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
