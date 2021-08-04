
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

#include <stdio.h>

#include "graphblas.hpp"

using namespace grb;

static constexpr size_t problem_size = 100000;

struct output {
	int exit_code;
	grb::utils::TimerResults times;
};

struct input {
	size_t n;
};

void grbProgram( const struct input & in, struct output & out ) {

	const size_t n = in.n;

	grb::Vector< int > x( n ), y( n );
	enum RC return_code = SUCCESS;

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

	int alpha = 0;
	return_code = grb::dot< grb::descriptors::no_operation, grb::Semiring< grb::operators::add< int >, grb::operators::mul< int >, grb::identities::zero, grb::identities::one > >( alpha, x, y );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::dot to calculate alpha = (x,y) returns bad error code (%d).\n", (int)return_code );
		out.exit_code = 3;
		return;
	}

	if( alpha != 2 * static_cast< int >( n ) ) {
		(void)fprintf( stderr,
			"Computed value by grb::dot (%d) does not equal expected value "
			"(%zd).\n",
			alpha, 2 * n );
		out.exit_code = 4;
		return;
	}

	out.exit_code = 0;
}

int main( int argc, char ** argv ) {
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	if( argc != 1 ) {
		(void)printf( "Usage: ./%s (this will attempt to run at MAX_P)\n", argv[ 0 ] );
		return EXIT_SUCCESS;
	}

	struct input in;
	in.n = problem_size;
	struct output out;

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
