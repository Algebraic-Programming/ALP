
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

#include <stdbool.h>

#include <iostream>

#include "graphblas/utils/parser.hpp"

#include "graphblas.hpp"


using namespace grb;

struct output {
	int exit_code;
};

struct input {
	char filename[ 1024 ];
	bool direct;
};

void grbProgram( const struct input &in, struct output &out ) {

	grb::utils::MatrixFileReader< double, size_t > parser( in.filename, in.direct );
	const size_t m = parser.m();
	const size_t n = parser.n();
	grb::Matrix< double > A( m, n );
	grb::Vector< double > x( m ), y( n );

	RC return_code = grb::buildMatrixUnique( A, parser.begin(), parser.end(), SEQUENTIAL );

	return_code = grb::set( x, 1 );
	if( return_code != SUCCESS ) {
		std::cerr << "grb::set (on x) returns bad error code (" << ((int)return_code) << ").\n";
		out.exit_code = 1;
		return;
	}

	return_code = grb::set( y, 2 );
	if( return_code != SUCCESS ) {
		std::cerr << "grb::set (on y) returns bad error code (" << ((int)return_code) << ").\n";
		out.exit_code = 2;
		return;
	}

	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> reals;

	return_code = grb::vxm< grb::descriptors::no_operation >( y, x, A, reals );
	if( return_code != SUCCESS ) {
		std::cerr << "grb::vxm returns bad error code (" << ((int)return_code) << ").\n";
		out.exit_code = 3;
		return;
	}

	const size_t P = grb::spmd<>::nprocs();
	const size_t s = grb::spmd<>::pid();
	if( s == 0 ) {
		std::cout << "%%MatrixMarket vector coordinate double general\n";
		std::cout << "%Global index \tValue\n";
		std::cout << grb::size( y ) << "\n";
	}
	for( size_t k = 0; k < P; ++k ) {
		if( k == s ) {
			for( const auto &pair : y ) {
				const size_t index = pair.first;
				std::cout << index << " "
					<< std::round( pair.second ) << "\n";
			}
		}
		grb::spmd<>::barrier();
	}

	out.exit_code = 0;

	std::cout << std::flush;
	std::cerr << std::flush;
}

int main( int argc, char ** argv ) {
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	if( argc != 2 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <matrix input file>" << std::endl;
		return EXIT_SUCCESS;
	}

	struct input in;
	struct output out;

	(void)strncpy( in.filename, argv[ 1 ], 1023 );
	in.filename[ 1023 ] = '\0';

	grb::Launcher< AUTOMATIC > automatic_launcher;

	if( automatic_launcher.exec( &grbProgram, in, out, true ) != SUCCESS ) {
		std::cout << "Test FAILED (launcher did not return SUCCESS).\n" << std::endl;
		return EXIT_FAILURE;
	}

	if( out.exit_code != 0 ) {
		std::cout << "Test FAILED (program returned non-zero exit code "
			<< out.exit_code << "\n" << std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}
	return out.exit_code;
}

