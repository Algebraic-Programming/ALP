
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

#include <cstdio>

#include <graphblas.hpp>
#include <graphblas/algorithms/matrix_factory.hpp>


using namespace grb;

static const double data1[ 15 ] = { 4.32, 7.43, 4.32, 6.54, 4.21, 7.65, 7.43, 7.54, 5.32, 6.43, 7.43, 5.42, 1.84, 5.32, 7.43 };
static const double data2[ 15 ] = { 8.49, 7.84, 8.49, 6.58, 8.91, 7.65, 7.84, 7.58, 5.49, 6.84, 7.84, 5.89, 1.88, 5.49, 7.84 };
static const double chk[ 15 ] = { 12.81, 15.27, 12.81, 13.12, 13.12, 15.30, 15.27, 15.12, 10.81, 13.27, 15.27, 11.31, 3.72, 10.81, 15.27 };

void alpProgram( const grb::RC &rc_in, int &error ) {
	assert( rc_in == grb::SUCCESS );
#ifdef NDEBUG
	(void) rc_in;
#endif
	RC rc = SUCCESS;

	// allocate
	grb::Vector< double > x( 15 );
	grb::Matrix< double > A = grb::factory::diag< double >( 15, 15, data2,
		data2 + 15 );
	grb::Vector< double > y( 15 );

	// initialise
	const double * const iterator = &( data1[ 0 ] );
	rc = grb::buildVector( x, iterator, iterator + 15, grb::SEQUENTIAL );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Unexpected return code from Vector build (x): "
			<< grb::toString( rc ) << ".\n";
		error = 4;
		return;
	}
	rc = grb::set( y, 1 );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Unexpected return code from Vector assign (y): "
			<< grb::toString( rc ) << ".\n";
		error = 5;
		return;
	}

	// get a semiring where multiplication is addition, and addition is
	// multiplication
	// this also tests if the proper identity is used
	typename grb::Semiring<
		grb::operators::mul< double >, grb::operators::add< double >,
		grb::identities::one, grb::identities::zero
	> switched;

	// execute what amounts to elementwise vector addition
	rc = grb::vxm( y, x, A, switched );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Unexpected return code from grb::vmx (y=xA): "
			<< grb::toString( rc ) << ".\n";
		error = 7;
		return;
	}

	// check
	for( const std::pair< size_t, double > &pair : y ) {
		if( !grb::utils::equals( chk[ pair.first ], pair.second, 1 ) ) {
			std::cerr << "Output vector element mismatch at position " << pair.first << ": "
				<< chk[ pair.first ] << " does not equal " << pair.second << ".\n";
			error = 8;
			break;
		}
	}

	// done
}

int main( int argc, char ** argv ) {
	(void) argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	// sanity check
	int error = 0;
	for( size_t i = 0; i < 15; ++i ) {
		if( !grb::utils::equals( data1[ i ] + data2[ i ], chk[ i ], 1 ) ) {
			std::cerr << "Sanity check error at position " << i << ": " << data1[ i ]
				<< " + " << data2[ i ] << " does not equal " << chk[ i ] << ".\n";
			error = 1;
		}
	}

	// initialise
	if( error == 0 ) {
		grb::RC rc = grb::SUCCESS;
		grb::Launcher< grb::AUTOMATIC > launcher;
		rc = launcher.exec( alpProgram, rc, error, true );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Could not launch the ALP program.\n";
			error = 10;
		}

		if( !error ) {
			std::cout << "Test OK\n" << std::endl;
		} else {
			std::cerr << std::flush;
			std::cout << "Test FAILED\n" << std::endl;
		}
	}

	// done
	return error;
}

