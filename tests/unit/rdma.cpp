
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
#include <mpi.h>

#include <graphblas.hpp>

const int LPF_MPI_AUTO_INITIALIZE = 0;

void grbProgram( const size_t &n, grb::RC &rc ) {
	const size_t s = grb::spmd<>::pid();
	const size_t P = grb::spmd<>::nprocs();

	assert( s < P );
	assert( P > 1 );

	if( s == 0 ){
		std::cout << "Testing RDMA of POD" << std::endl;
	}
	int x = 42, y = 69;

	if( s == 0 ){
		std::cout << "\t register" << std::endl;
	}
	rc = rc ? rc : grb::rdma< >::register_global( x );
	assert( rc == grb::SUCCESS );

	if( s == 0 ){
		std::cout << "\t put" << std::endl;
	}
	if( s == 0 ){
		rc = rc ? rc : grb::rdma< >::put( y, 1, x );
		x = 0;
	}
	rc = rc ? rc : grb::spmd<>::sync();
	assert( rc == grb::SUCCESS );
	assert( s != 1 || x == 69 );

	rc = rc ? rc : grb::spmd<>::sync();

	if( s == 0 ){
		std::cout << "\t get" << std::endl;
	}
	if( s == 1 ){
		rc = rc ? rc : grb::rdma< >::get( 0, x, y );
	}
	rc = rc ? rc : grb::spmd<>::sync();
	assert( s != 1 || y == 0 );
	rc = rc ? rc : grb::spmd<>::sync();

	if( rc != grb::SUCCESS ) return;
	if( s == 0 ){
		std::cout << "Testing RDMA of grb::Vector<reference>" << std::endl;
	}

	using T=float;
	grb::Vector< T, grb::reference > a ( n );
	grb::Vector< T, grb::reference > b ( n );
	constexpr T v1 = 42.69;

	if( s == 0 ){
		std::cout << "\t register" << std::endl;
	}

	rc = rc ? rc : grb::set( a, static_cast<T>( s ) );
	rc = rc ? rc : grb::set( b, static_cast<T>( 0 ) );

	rc = rc ? rc : grb::rdma<>::register_global( a );
	assert( rc == grb::SUCCESS );
	rc = rc ? rc : grb::spmd<>::sync();

	rc = rc ? rc : grb::rdma<>::localRegisterSize( 2 );

	if( s == 0 ){
		std::cout << "\t put" << std::endl;
	}
	if( s == 0 ){
		rc = rc ? rc : grb::rdma<>::put( a, 1, a );
	}
	rc = rc ? rc : grb::spmd<>::sync();
	assert( rc == grb::SUCCESS );

	if( s == 1 ){
		for(const auto &i : a ){
			assert( i.second == static_cast< T >( 0 ) );
		}
	}

	rc = rc ? rc : grb::rdma<>::localRegisterSize( 2 );
	rc = rc ? rc : grb::spmd<>::sync();
	assert( rc == grb::SUCCESS );
	if( s == 0 ){
		std::cout << "\t get" << std::endl;
		rc = rc ? rc : grb::set( a, v1 );
	}
	rc = rc ? rc : grb::spmd<>::sync();
	if( s == 1 ){
		rc = rc ? rc : grb::rdma<>::get( 0, a, b );
	}
	rc = rc ? rc : grb::spmd<>::sync();
	assert( rc == grb::SUCCESS );

	if( s == 1 ){
		sleep(1);
		for(const auto &i : b ){
			assert( i.second == v1 );
		}
	}

	return;
}

int main( int argc, char ** argv ) {
	(void) argc;
	size_t n = 42;
	grb::RC out = grb::SUCCESS;

	if( MPI_Init( &argc, &argv ) != MPI_SUCCESS ) {
		std::cerr << "MPI_Init returns with non-SUCCESS exit code." << std::endl;
		return 10;
	}

	std::cout << "This is a functional test " << argv[ 0 ] << "\n";
	grb::Launcher< grb::FROM_MPI > launcher;
	if( launcher.exec( &grbProgram, n, out, true ) != grb::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}

	if( MPI_Finalize() != MPI_SUCCESS ) {
		std::cerr << "MPI_Finalize returns with non-SUCCESS exit code." << std::endl;
		return 50;
	}

	if( out == grb::SUCCESS ){
		std::cerr << "Test OK\n";
	}else{
		std::cerr << "Test FAILED\n";
	}

}

