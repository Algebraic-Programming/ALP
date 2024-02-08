
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
static const int M_val[ 2 ] = { 1, 1 };

static const size_t M_I[ 2 ] = { 8, 9 };
static const size_t M_J[ 2 ] = { 9, 8 };

// graphblas program
void grbProgram( const void *, const size_t in_size, int &error ) {
	error = 0;

	if( in_size != 0 ) {
		(void) fprintf( stderr, "Unit tests called with unexpected input\n" );
		error = 1;
		return;
	}

	// allocate
	grb::Matrix< int > M( 10, 10 );

	grb::RC rc = buildMatrixUnique( M, &( M_I[ 0 ] ), &( M_J[ 0 ] ), M_val, 2, SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t initial buildMatrixUnique FAILED\n";
		error = 5;
	}

	if( !error ) {
		rc =  grb::eWiseLambda( [&M]( const size_t i, const size_t j, int& nz ) {
				(void) i;
				nz = j;
			}, M );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t eWiseLambda call failed\n";
		error = 10;
	}

	if( !error ) {
		for( const auto &it : M ) {
			if( static_cast< size_t >(it.second) != it.first.second ) {
				std::cerr << "\t eWiseLambda returned incorrect result\n";
				error = 15;
				break;
			}
		}
	}
}

int main( int argc, char ** argv ) {
	(void) argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	int error;
	grb::Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, nullptr, 0, error ) != SUCCESS ) {
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

