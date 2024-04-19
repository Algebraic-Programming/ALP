
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

#include <utility>
#include <iostream>

#include "graphblas.hpp"


using namespace grb;

// sample data
static const double vec1_vals[ 3 ] = { 1, 2, 3 };
static const double vec2_vals[ 3 ] = { 4, 5, 6 };

static const size_t I[ 3 ] = { 0, 1, 2 };

static const double test1_in[ 3 ] = { 1, 1, 1 };
static const double test1_expect[ 3 ] = { 24, 30, 36 };

static const double test2_in[ 3 ] = { 1, 1, 1 };
static const double test2_expect[ 3 ] = { 15, 30, 45 };

// graphblas program
void grbProgram( const void *, const size_t in_size, int &error ) {
	/** \internal TODO: Implement initialization and result checking.
	 * Currently only serves as the interface showcase.
	 * */
	error = 0;

	if( in_size != 0 ) {
		(void)fprintf( stderr, "Unit tests called with unexpected input\n" );
		error = 1;
		return;
	}

	// allocate
	grb::VectorView< double > u( 3 );
	grb::VectorView< double > v( 3 );
	grb::StructuredMatrix< double, structures::General > M( 3, 3 );
	// grb::Vector< double > test1( 3 );
	// grb::Vector< double > out1( 3 );
	// grb::Vector< double > test2( 3 );
	// grb::Vector< double > out2( 3 );

	// semiring
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	grb::RC rc;

	// initialise vec
	// const double * vec_iter = &(vec1_vals[ 0 ]);
	// grb::RC rc = grb::buildVector( u, vec_iter, vec_iter + 3, SEQUENTIAL );
	// if( rc != SUCCESS ) {
	// 	std::cerr << "\t initial buildVector FAILED\n";
	// 	error = 5;
	// }

	// if( !error ) {
	// 	vec_iter = &(vec2_vals[ 0 ]);
	// 	rc = grb::buildVector( v, vec_iter, vec_iter + 3, SEQUENTIAL );
	// }
	// if( rc != SUCCESS ) {
	// 	std::cerr << "\t initial buildVector FAILED\n";
	// 	error = 10;
	// }

	if( !error ) {
		rc = grb::outer( M, u, v, ring.getMultiplicativeOperator(), SYMBOLIC );
		rc = rc ? rc : grb::outer( M, u, v, ring.getMultiplicativeOperator() );
	}

}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	int error;
	grb::Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, error ) != SUCCESS ) {
		std::cerr << "Test failed to launch\n";
		error = 255;
	}
	if( error == 0 ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
	}

	// done
	return error;
}

