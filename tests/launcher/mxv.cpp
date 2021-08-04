
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

#include <stdbool.h>
#include <stdio.h>

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

void grbProgram( const struct input & in, struct output & out ) {

	grb::utils::MatrixFileReader< double, size_t > parser( in.filename, in.direct );
	const size_t m = parser.m();
	const size_t n = parser.n();
	grb::Matrix< double > A( m, n );
	grb::Vector< int > x( n ), y( m );

	RC return_code = grb::buildMatrixUnique( A, parser.begin(), parser.end(), SEQUENTIAL );

	return_code = grb::set( x, 1 );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::set (on x) returns bad error code (%d).\n", (int)return_code );
		out.exit_code = 1;
		return;
	}

	return_code = grb::set( y, 2 );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::set (on y) returns bad error code (%d).\n", (int)return_code );
		out.exit_code = 2;
		return;
	}

	grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > reals;
	return_code = grb::mxv< grb::descriptors::no_operation >( y, A, x, reals );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::mxv returns bad error code (%d).\n", (int)return_code );
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
			for( const auto pair : y ) {
				const size_t index = pair.first;
				std::cout << index << " "
						  << " " << pair.second << "\n";
			}
		}
		grb::spmd<>::barrier();
	}

	out.exit_code = 0;
}

int main( int argc, char ** argv ) {
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	if( argc != 2 ) {
		(void)printf( "Usage: ./%s <matrix input file>\n", argv[ 0 ] );
		return EXIT_SUCCESS;
	}

	struct input in;
	struct output out;

	(void)strncpy( in.filename, argv[ 1 ], 1023 );
	in.filename[ 1023 ] = '\0';

	grb::Launcher< AUTOMATIC > automatic_launcher;

	if( automatic_launcher.exec( &grbProgram, in, out ) != SUCCESS ) {
		(void)printf( "Test FAILED (launcher did not return SUCCESS).\n\n" );
		return EXIT_FAILURE;
	}

	if( out.exit_code != 0 ) {
		(void)printf( "Test FAILED (program returned non-zero exit code "
					  "%d)\n\n",
			out.exit_code );
	} else {
		(void)printf( "Test OK.\n\n" );
	}

	return out.exit_code;
}
