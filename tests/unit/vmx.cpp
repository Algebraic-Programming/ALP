
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

#include <graphblas.hpp>
#include <graphblas/algorithms/matrix_factory.hpp>


using namespace grb;
using namespace grb::algorithms;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const int data2[ 15 ] = { 8, 9, 8, 6, 8, 7, 8, 7, 5, 2, 3, 5, 1, 5, 5 };
static const int chk[ 15 ] = { 32, 63, 32, 36, 32, 49, 8, 49, 15, 12, 21, 25, 1, 40, 35 };

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	// sanity check against metabugs
	int error = 0;
	for( size_t i = 0; i < 15; ++i ) {
		if( !grb::utils::equals( data1[ i ] * data2[ i ], chk[ i ] ) ) {
			std::cerr << "Sanity check error at position " << i << ": "
				<< data1[ i ] << " + " << data2[ i ] << " "
				<< "does not equal " << chk[ i ] << "."
				<< std::endl;
			error = 1;
		}
	}

	// initialise
	enum grb::RC rc = grb::init();
	if( rc != grb::SUCCESS ) {
		std::cerr << "Unexpected return code from grb::init: "
			<< static_cast< int >(rc) << "." << std::endl;
		error = 2;
	}

	// exit early if failure detected as this point
	if( error ) {
		std::cout << "Test FAILED\n" << std::endl;
		return error;
	}

	// allocate
	grb::Vector< int > x( 15 );
	grb::Vector< int > y( 15 );
	grb::Matrix< int > A = matrices< int >::diag( 15, 15, data2, data2 + 15 );

	// initialise x
	if( !error ) {
		const int * iterator = &( data1[ 0 ] );
		rc = grb::buildVector( x, iterator, iterator + 15, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from Vector build (x): "
			       << static_cast< int >(rc) << "." << std::endl;
			error = 4;
		}
	}

	// initialise y
	if( !error ) {
		rc = grb::set( y, 0 );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from Vector build (y): "
				<< static_cast< int >(rc) << "." << std::endl;
			error = 5;
		}
	}

	// check contents of x
	for( const std::pair< size_t, int > &pair : x ) {
		if( !grb::utils::equals( data1[ pair.first ], pair.second ) ) {
			std::cerr << "Initialisation error: vector x, "
				<< "element at position " << pair.first << ": "
				<< pair.second << " does not equal "
				<< data1[ pair.first ] << "." << std::endl;
			error = 20;
			break;
		}
	}

	// check contents of y
	for( const std::pair< size_t, int > &pair : y ) {
		if( !grb::utils::equals( 0, pair.second ) ) {
			std::cerr << "Initialisation error: vector y, "
			       << "element at position " << pair.first << ": "
			       << "0 does not equal " << pair.second
			       << "." << std::endl;
			error = 6;
			break;
		}
	}

	// get a semiring where multiplication is addition, and addition is multiplication
	// this also tests if the proper identity is used
	typename grb::Semiring<
		grb::operators::add< int >, grb::operators::mul< int >,
		grb::identities::zero, grb::identities::one
	> integers;

	// execute what amounts to elementwise vector addition
	if( !error ) {
		rc = grb::vxm( y, x, A, integers );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from grb::vxm: "
				<< static_cast< int >(rc) << "." << std::endl;
			error = 8;
		}
	}

	// check
	for( const std::pair< size_t, int > &pair : y ) {
		if( !grb::utils::equals( chk[ pair.first ], pair.second ) ) {
			std::cerr << "Output vector element mismatch at position "
				<< pair.first << ": " << chk[ pair.first ] << "does not equal "
				<< pair.second << "." << std::endl;
			error = 9;
			break;
		}
	}

	// finalize
	if( error ) {
		(void)grb::finalize();
	} else {
		rc = grb::finalize();
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from grb::finalize: "
				<< static_cast< int >(rc) << "." << std::endl;
			error = 10;
		}
	}

	if( !error ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cout << "Test FAILED\n" << std::endl;
	}

	// done
	return error;
}

