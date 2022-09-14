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

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include <alp.hpp>
#include <alp/utils/parser/MatrixFileReader.hpp>

using namespace alp;

std::string fname;

void alp_program( const size_t & in, alp::RC & rc ) {
	(void)in;
	rc = SUCCESS;

	alp::utils::MatrixFileReader<
		double
	> parser_A( fname );

	for ( auto it = parser_A.begin() ; it != parser_A.end() ; ++it  ) {
		std::cout << " i,j,v= " << it.i() << " " << it.j() << " " << it.v() << "\n";
	}

	return;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		std::istringstream ss( argv[ 1 ] );
		if( ! ( ss >> fname ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( ! ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else {
			// all OK
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " \n";
		std::cerr << "  -filename \n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	alp::Launcher< AUTOMATIC > launcher;
	alp::RC out;
	size_t in;
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
