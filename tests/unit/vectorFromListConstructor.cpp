
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
#include <vector>

#include "graphblas.hpp"

using namespace grb;

void grbProgram( const void *, const size_t in_size, int & error ) {
	error = 0;

	if( in_size != 0 ) {
		std::cerr << "Unit tests called with unexpected input" << std::endl;
		error = 1;
		return;
	}
	std::vector< int > values = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
	grb::Vector< int > x( { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 } );
	bool equals = true;
	auto vector_it = x.begin();
	for( size_t i = 0; i < values.size(); i++ ) {
		auto position = vector_it->first;
		auto value = vector_it->second;
		if( i != position || values[ i ] != value ) {
			std::cerr << "Expected position " << i << " value " << values[ i ] << " but got position " << position << " value " << value << std::endl;
			equals = false;
			break;
		}
		vector_it.operator++();
	}
	if( !equals ) {
		std::cerr << "Vector values are not correct" << std::endl;
		error = 1;
		return;
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << std::endl;

	int error;
	grb::Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, error ) != SUCCESS ) {
		std::cout << "Test FAILED (test failed to launch)" << std::endl;
		error = 255;
	}
	if( error == 0 ) {
		std::cout << "Test OK" << std::endl;
	} else {
		std::cout << "Test FAILED" << std::endl;
	}

	// done
	return error;
}

