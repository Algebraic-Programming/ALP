
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
#include <graphblas/algorithms/matrix_factory.hpp>

using namespace grb;

using namespace grb::algorithms;

void grb_program( const int &, grb::RC &rc ) {

	// large non-square mixed-domain matrix check
	{
		Matrix< char > A = matrices< char >::identity( 10000000, 2000000, 2 );
		Matrix< float > B = matrices< float >::identity( 10000000, 2000000, 2 );
		Matrix< size_t > C = matrices< size_t >::identity( 10000000, 2000000, 2 );

		rc = grb::eWiseApply( C, A, B,
			grb::operators::add< float, size_t, char >(), RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A, B,
			grb::operators::add< float, size_t, char >() );
		if( rc != SUCCESS ) {
			std::cout << "Error on executing large non-square "
				<< "mixed-domain matrix check\n";
			return;
		}
		for( const auto &triple : C ) {
			const size_t &i = triple.first.first;
			const size_t &j = triple.first.second;
			const size_t &v = triple.second;
			if( i != j ) {
				std::cout << "Unexpected entry at position ( " << i << ", " << j << " ) "
					<< "-- only expected entries on the diagonal\n";
				rc = FAILED;
			}
			if( v != 4 ) {
				std::cout << "Unexpected value at position ( " << i << ", " << j << " ) "
					<< "= " << v << " -- expected 4\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		std::cout << "Error detected in large non-square mixed-domain matrix check "
			<< "-- exiting\n";
		return;
	}
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	int input = 0; // unused

	// error checking
	if( argc > 1 ) {
		printUsage = true;
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << "\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, input, out, false ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}

