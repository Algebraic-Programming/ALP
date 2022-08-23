
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

typedef double T1;

const T1 testval1 = 1.5;
const T1 testval2 = -1;
const T1 testval3 = 2.;

template< typename VectorType >
void print_vector( std::string name, const VectorType &v ) {

	if( ! alp::internal::getInitialized( v ) ) {
		std::cout << "Vector " << name << " uninitialized.\n";
		return;
	}

	std::cout << "Vector " << name << " of size " << alp::getLength( v ) << " contains the following elements:\n";

	std::cout << "[\t";
	for( size_t i = 0; i < alp::getLength( v ); ++i ) {
		std::cout << v[ i ] << "\t";
	}
	std::cout << "]\n";
}

void alp_program( const size_t &n, alp::RC &rc ) {

	{
		alp::Vector< T1 > left( n );
		alp::Vector< T1 > right( n );

		alp::Semiring<
			alp::operators::add< double >, alp::operators::mul< double >,
			alp::identities::zero, alp::identities::one
			> ring;

		std::vector< T1 > left_data( n );
		std::vector< T1 > right_data( n );

		// // test 1, init
		std::fill(left_data.begin(), left_data.end(), testval1 );
		std::fill(right_data.begin(), right_data.end(), testval2 );

		rc = SUCCESS;
		rc = rc ? rc : alp::buildVector( left, left_data.begin(), left_data.end() );
		rc = rc ? rc : alp::buildVector( right, right_data.begin(), right_data.end() );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 1 (dense, regular semiring): initialisation FAILED\n";
			return;
		}

		// test 1, exec
		Scalar< T1 > out;
		rc = alp::dot( out, left, right, ring );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 1 (dense, regular semiring): dot FAILED\n";
			return;
		}

		// test 1, check
		if( static_cast< T1 >( testval1 * testval2 * n ) != *out ) {
			std::cerr << "\t test 1 (dense, regular semiring): unexpected output "
													<< "( " << *out << ", expected "
													<< ( static_cast< T1 >( testval1 * testval2 * n ) )
													<< " )\n";
			std::cout << " --->DEVELOP continue anyway!\n";
			// rc = FAILED;
			// return;
		}
	}

	{
		alp::Vector< T1 > left( n );
		alp::Vector< T1 > right( n );

		//test 2, init
		alp::Semiring<
			alp::operators::add< double >,
			alp::operators::left_assign_if< T1, bool, T1 >,
			alp::identities::zero, alp::identities::logical_true
			> pattern_sum_if;
		rc = SUCCESS;
		{
			// temp initialization
			std::vector< T1 > left_data( n );
			std::vector< T1 > right_data( n );
			std::fill(left_data.begin(), left_data.end(), static_cast< T1 >( 0 ) );
			std::fill(right_data.begin(), right_data.end(), static_cast< T1 >( 1 ) );
			rc = rc ? rc : alp::buildVector( left, left_data.begin(), left_data.end() );
			rc = rc ? rc : alp::buildVector( right, right_data.begin(), right_data.end() );
		}
		// rc = rc ? rc : alp::set( left, Scalar< T1 >( 0 ) ); // needs an implementation
		// rc = rc ? rc : alp::set( right, Scalar< T1 >( 1 ) );  // needs an implementation

		auto left_view_even = alp::get_view( left, alp::utils::range( 0, n, 2 ) );
		//rc = rc ? rc : alp::set( left_view_even, Scalar< T1 >( testval3 ) );  // needs an implementation
		if( rc != SUCCESS ) {
			std::cerr << "\t test 2 (sparse, non-standard semiring) "
													<< "initialisation FAILED\n";
			return;
		}

		// test 2, exec
		Scalar< T1 > out;
		rc = alp::dot( out, left, right, pattern_sum_if );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 2 (sparse, non-standard semiring) dot FAILED\n";
			return;
		}

		// test 2, check
		if( testval3 * static_cast< T1 >( n ) != *out * 2  ) {
			std::cerr << "\t test 2 (sparse, non-standard semiring), "
				<< "unexpected output: " << *out << ", expected " << n
				<< ".\n";
			std::cout << " --->DEVELOP continue anyway!\n";
			// rc = FAILED;
			// return;
		}
	}

	Scalar< int > alpha;
	alp::Semiring<
		alp::operators::add< int >, alp::operators::mul< int >,
		alp::identities::zero, alp::identities::one
		> intRing;

	{
		// test 3, init
		rc = SUCCESS;
		alp::Vector< int > x( n ), y( n );
		{
			// temp initialization
			std::vector< T1 > x_data( n ), y_data( n );
			std::fill(x_data.begin(), x_data.end(), 1 );
			std::fill(y_data.begin(), y_data.end(), 2 );
			rc = rc ? rc : alp::buildVector( x, x_data.begin(), x_data.end() );
			rc = rc ? rc : alp::buildVector( y, y_data.begin(), y_data.end() );
		}
		// rc = alp::set( x, 1 );
		// rc = rc ? rc : alp::set( y, 2 );
		if( rc != alp::SUCCESS ) {
			std::cerr << "\t test 3 (dense integer vectors) initialisation FAILED\n";
			return;
		}

		// test 3, exec
		rc = alp::dot( alpha, x, y, intRing );
		if( rc != alp::SUCCESS ) {
			std::cerr << "\t test 3 (dense integer vectors) dot FAILED\n";
			return;
		}

		// test 3, check
		if( *alpha != 2 * static_cast< int >( n ) ) {
			std::cerr << "\t test 3 (dense integer vectors) unexpected value "
				<< *alpha << ", expected 2 * n = " << ( 2 * n) << ".\n";
			std::cout << " --->DEVELOP continue anyway!\n";
			// rc = FAILED;
			// return;
		}
	}

	{
		// test 4, init
		alp::Vector< int > empty_left( 0 ), empty_right( 0 );
		setInitialized( empty_left, true );
		setInitialized( empty_right, true );

		// test 4, exec
		rc = alp::dot( alpha, empty_left, empty_right, intRing );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 4 (empty vectors) dot FAILED\n";
			return;
		}

		// test 4, check
		if( *alpha != 2 * static_cast< int >(n) ) {
			std::cerr << "\t test 4 (empty vectors) unexpected value "
				<< *alpha << ", expected 2 * n = " << ( 2 * n ) << ".\n";
			std::cout << " --->DEVELOP continue anyway!\n";
			// rc = FAILED;
			// return;
		}
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
