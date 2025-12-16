
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
#include <unistd.h>

#include <graphblas.hpp>

void grbProgram( const size_t &in, grb::RC &exit_status ) {
	(void) in;
	grb::RC rc = grb::SUCCESS;

	const size_t s = grb::spmd<>::pid();
	const size_t P = grb::spmd<>::nprocs();
	assert( s < P );
	assert( P > 1 );
	// sleep( 10 );
	int x = 42, y = 69;

	rc = rc ? rc : grb::rdma< >::register_global( x );
	// rc = rc ? rc : grb::rdma< >::register_global( y ); // sus

	assert( rc == grb::SUCCESS );
	if( s == 0 ){
		rc = rc ? rc : grb::rdma< >::put( y, 1, x );
		x = 0;
	}
	assert( rc == grb::SUCCESS );
	rc = rc ? rc : grb::spmd<>::sync();
	assert( rc == grb::SUCCESS );
	assert( s != 1 || x == 69 );

	rc = rc ? rc : grb::spmd<>::sync();

	if( s == 1 ){
		rc = rc ? rc : grb::rdma< >::get( 0, x, y );
	}
	assert( rc == grb::SUCCESS );
	rc = rc ? rc : grb::spmd<>::sync();
	assert( rc == grb::SUCCESS );
	assert( s != 1 || y == 0 );

	assert( rc == grb::SUCCESS );
	exit_status = rc;
	return;
}

int main( int argc, char ** argv ) {
	(void) argc;
	size_t in = 42;
	grb::RC out;

	std::cout << "This is a functional test " << argv[ 0 ] << "\n";
	grb::Launcher< grb::AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, in, out, true ) != grb::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}

	if( out == grb::SUCCESS ){
		std::cerr << "Test OK\n";
	}else{
		std::cerr << "Test FAILED\n";
	}
}

