
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

#include <cstdio>

#define USE1 (void)printf( "Usage: %s\n", argv[ 0 ] );

#include <lpf/core.h>

void spmd( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args );

int main( int argc, char ** argv ) {

	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	if( argc > 1 ) {
		USE1 return 0;
	}

	// prepare args
	int exit_status = 0;
	bool automatic = true;
	lpf_args_t args = { &automatic, sizeof( bool ), &exit_status, sizeof( int ), NULL, 0 };

	// call SPMD
	spmd( NULL, 0, 1, args );

	// master process reports test success
	if( exit_status ) {
		(void)printf( "Test FAILED.\n\n" );
		(void)fflush( stdout );
		(void)fflush( stderr );
		return exit_status;
	}

	(void)printf( "Test OK.\n\n" );
	(void)fflush( stdout );
	return 0;
}

#undef USE1
