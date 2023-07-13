
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
	(void)n;

	{ // Test: logical_and< bool > just to make sure it works
		bool value = true;
		rc = rc ? rc : foldl( value, true, operators::logical_and< bool >() );

		// Check results
		if( value != true || rc != SUCCESS ) {
			rc = FAILED;
			std::cerr << "Test logical_and< bool > FAILED\n";
			return;
		}
	}

	{ // Test: logical_not< logical_and< bool > >
		bool value = true;
		rc = rc ? rc : foldl( value, true, operators::logical_not< operators::logical_and< bool > >() );

		// Check results
		if( value != false || rc != SUCCESS ) {
			rc = FAILED;
			std::cerr << "Test logical_not< logical_and< bool > > FAILED\n";
			return;
		}
	}

	// done
	return;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << "\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out = SUCCESS;
	size_t unused = 0;
	if( launcher.exec( &grb_program, unused, out, true ) != SUCCESS ) {
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
