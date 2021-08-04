
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

#include "graphblas.hpp"

using namespace grb;

// a collection of tests on empty vectors. If there are multiple failures, the
// return code will reflect the last encountered failure.

int main( int argc, char ** argv ) {
	(void)argc;
	int fail = 0; // assume success by default
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	enum grb::RC rc = grb::init();
	if( rc != SUCCESS ) {
		(void)printf( "grb::init returns non-SUCCESS code %d.\n", (int)rc );
		return 10;
	}

	grb::Vector< int > test( 0 );

	if( grb::size( test ) != 0 ) {
		(void)printf( "grb::size should return zero (got %zd instead)\n", grb::size( test ) );
		fail = 1;
	}

	if( grb::set( test, 1 ) != grb::SUCCESS ) {
		(void)printf( "grb::set (all elements) returns non-SUCCESS code\n" );
		fail = 2;
	}

	if( grb::setElement( test, 1, 0 ) != grb::MISMATCH ) {
		(void)printf( "grb::set (one element at index 0) returns non-MISMATCH "
					  "code\n" );
		fail = 3;
	}

	rc = grb::finalize();
	if( rc != SUCCESS ) {
		(void)printf( "grb::finalize returns non-SUCCESS code %d.\n", (int)rc );
		return 20;
	}

	if( fail ) {
		(void)printf( "Test FAILED.\n" );
	} else {
		(void)printf( "Test OK.\n\n" );
	}
#ifndef _GRB_NO_STDIO
	(void)fflush( stdout );
	(void)fflush( stderr );
#endif
	return fail;
}
