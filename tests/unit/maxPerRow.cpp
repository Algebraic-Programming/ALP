
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

#include <graphblas.hpp>

#include <utils/matrix_values_check.hpp>


using namespace grb;

// sample data
static const int Val_input[ 9 ] = { 3, 3, 1, 5, 6, 2, 3, 4, 3 };

static const size_t I_input[ 9 ] = { 0, 0, 0, 1, 3, 3, 4, 4, 4 };
static const size_t J_input[ 9 ] = { 0, 1, 2, 2, 2, 4, 1, 2, 4 };

static const int Val_output[ 4 ] = { 3, 4, 6, 5 };

static const size_t I_output[ 4 ] = { 0, 4, 3, 1 };
static const size_t J_output[ 4 ] = { 1, 2, 2, 2 };

// graphblas program
void grbProgram( const void *, const size_t in_size, int &error ) {
	error = 0;

	if( in_size != 0 ) {
		(void)fprintf( stderr, "Unit tests called with unexpected input\n" );
		error = 1;
		return;
	}

	// allocate
	grb::Matrix< int > Input( 5, 5 );
	grb::Matrix< int > Output( 5, 5 );
	grb::Matrix< int > ExpectedOutput( 5, 5 );

	grb::RC rc = grb::resize( Input, 9 );

	if( rc != SUCCESS ) {
		std::cerr << "\t initial input resize FAILED\n";
		error = 5;
	}

	if( !error ) {
		rc = grb::buildMatrixUnique( Input, &( I_input[ 0 ] ), &( J_input[ 0 ] ), Val_input, 9, SEQUENTIAL );
	}

	if( rc != SUCCESS ) {
		std::cerr << "\t initial input build FAILED\n";
		error = 10;
	}

	if( !error ) {
		rc = grb::resize( ExpectedOutput, 4 );
	}

	if( rc != SUCCESS) {
		std::cerr << "\t expected output resize FAILED\n";
		error = 15;
	}

	if( !error ) {
		rc = grb::buildMatrixUnique( ExpectedOutput, &( I_output[ 0 ] ), &( J_output[ 0 ] ), Val_output, 4, SEQUENTIAL );
	}

	if( rc != SUCCESS) {
		std::cerr << "\t expected output build FAILED\n";
		error = 20;
	}

	if( !error ) {
		rc = grb::internal::maxPerRow( Output, Input, Phase::RESIZE );
	}

	if( rc != SUCCESS ) {
		std::cerr << "\t maxPerRow resize FAILED\n";
		error = 25;
	}

	if( !error ) {
		rc = grb::internal::maxPerRow( Output, Input, Phase::EXECUTE );
	}

	if( rc != SUCCESS ) {
		std::cerr << "\t maxPerRow execution FAILED\n";
		error = 30;
	}

	if( !error ) {
		if( utils::compare_crs( ExpectedOutput, Output ) != SUCCESS ) {
			std::cerr << "\t unexpected CRS of output\n";
			error = 35;
		}
	}

	if( !error ) {
		if( grb::utils::compare_ccs( ExpectedOutput, Output ) != SUCCESS ) {
			std::cerr << "\t unexpected CCS of output\n";
			error = 40;
		}
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
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

