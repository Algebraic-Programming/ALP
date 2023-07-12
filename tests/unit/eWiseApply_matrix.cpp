
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

	// large non-square mixed-domain matrix check
	{
		grb::Matrix< char > A( n, 2*n );
		grb::Matrix< float > B( n, 2*n );
		grb::Matrix< size_t > C( n, 2*n );
		size_t * I = new size_t[ n ];
		size_t * J = new size_t[ n ];
		char * V = new char[ n ];
		for( size_t k = 0; k < n; ++k ) {
			I[ k ] = k;
			J[ k ] = k+n;
			V[ k ] = 2;
		}
		rc = grb::buildMatrixUnique( A, I, J, V, n, SEQUENTIAL );
		rc = rc ? rc : grb::buildMatrixUnique( B, I, J, V, n, SEQUENTIAL );
		rc = rc ? rc : grb::buildMatrixUnique( C, I, J, V, n, SEQUENTIAL );
		rc = rc ? rc : grb::eWiseApply( C, A, B,
			grb::operators::add< float, size_t, char >(), RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A, B,
			grb::operators::add< float, size_t, char >() );
		if( rc != SUCCESS ) {
			std::cout << "Error on executing large non-square "
				<< "mixed-domain matrix check\n";
			return;
		}
		
		for( const auto &triple : C ) {
			const auto &i = triple.first.first;
			const auto &j = triple.first.second;
			const auto &v = triple.second;
			if( j != i+n ) {
				std::cout << "Unexpected entry at position ( " << i << ", " << i+n << " ) "
					<< "-- only expected entries on the n-th diagonal\n";
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
	size_t input = 1000; // unused

	// error checking
	if( argc > 1 ) {
		input = std::strtoul( argv[ 1 ], nullptr, 10 );
	}
	if( argc > 2 ) {
		std::cerr << "Usage: " << argv[ 0 ] << "[n]\n";
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

