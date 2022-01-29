
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

#include <assert.h>

#include "graphblas.hpp"


using namespace grb;

// a collection of tests on empty vectors. If there are multiple failures, the
// return code will reflect the last encountered failure.

void grbProgram( const int &, int &fail ) {
	assert( fail == 0 );

	grb::Vector< int > test( 0 );

	if( grb::size( test ) != 0 ) {
		std::cerr << "grb::size should return zero; got "
			<< grb::size( test ) << " instead"
			<< std::endl;
		fail += 1;
	}

	if( grb::set( test, 1 ) != grb::SUCCESS ) {
		std::cerr << "grb::set (all elements) returns non-SUCCESS code" << std::endl;
		fail += 10;
	}

	if( grb::setElement( test, 1, 0 ) != grb::MISMATCH ) {
		std::cerr << "grb::set (one element at index 0) "
			<< "does not return MISMATCH" << std::endl;
		fail += 100;
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << std::endl;

	int fail = 0; // assume success by default
	grb::Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, fail, fail ) != grb::SUCCESS ) {
		std::cout << "Test FAILED (launcher did not return SUCCESS)\n" << std::endl;
		return 255;
	}

	if( fail ) {
		std::cout << "Test FAILED.\n" << std::endl;
	} else {
		std::cout << "Test OK.\n" << std::endl;
	}
	return fail;
}

