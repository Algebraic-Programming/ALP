
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
#include <sstream>

#include <graphblas/algorithms/matrix_factory.hpp>

#include <graphblas.hpp>

using namespace grb;

void grb_program( const size_t & n, grb::RC & rc ) {

	{ // grb::factory::identity<void>
		Matrix< void > I = factory::identity< void >( n, IOMode::SEQUENTIAL );
		size_t i = 0;
		for( const auto & e : I ) {
			if( e.first != e.second || e.first != i++ ) {
				std::cerr << "Test FAILED: grb::factory::identity<void> (sequential)\n";
				rc = FAILED;
				return;
			}
		}
	}

	{ // grb::factory::identity<int>
		Matrix< int > I = factory::identity< int >( n, IOMode::SEQUENTIAL, 2 );
		size_t i = 0;
		for( const auto & e : I ) {
			if( e.first.first != e.first.second || e.first.first != i++ ) {
				std::cerr << "Test FAILED: grb::factory::identity<int> (sequential): incorrect coordinate\n";
				rc = FAILED;
				return;
			}
			if( e.second != 2 ) {
				std::cerr << "Test FAILED: grb::factory::identity<int> (sequential): incorrect value\n";
				rc = FAILED;
				return;
			}
		}
	}

	{ // grb::factory::dense<int> of size: [n,n]
		Matrix< int > D = factory::dense< int >( n, n, IOMode::SEQUENTIAL, 2 );
		std::vector< bool > touched( n * n, false );
		for( const auto & e : D ) {
			touched[ e.first.first * nrows( D ) + e.first.second ] = true;
			if( e.second != 2 ) {
				std::cerr << "Test FAILED: grb::factory::identity<int>[n,n]: incorrect value\n";
				rc = FAILED;
				return;
			}
		}
		if( std::find( touched.begin(), touched.end(), false ) != touched.end() ) {
			std::cerr << "Test FAILED: grb::factory::identity<int>[n,n]: not dense\n";
			rc = FAILED;
			return;
		}
	}

	{ // grb::factory::dense<int> of size: [1,n]
		Matrix< int > D = factory::dense< int >( 1, n, IOMode::SEQUENTIAL, 2 );
		std::vector< bool > touched( n, false );
		for( const auto & e : D ) {
			touched[ e.first.second ] = true;
			if( e.first.first != 0 ) {
				std::cerr << "Test FAILED: grb::factory::identity<int>[1,n]: incorrect row coordinate\n";
				rc = FAILED;
				return;
			}
			if( e.second != 2 ) {
				std::cerr << "Test FAILED: grb::factory::identity<int>[1,n]: incorrect value\n";
				rc = FAILED;
				return;
			}
		}
		if( std::find( touched.begin(), touched.end(), false ) != touched.end() ) {
			std::cerr << "Test FAILED: grb::factory::identity<int>[1,n]: not dense\n";
			rc = FAILED;
			return;
		}
	}

	{ // grb::factory::dense<int> of size: [n,1]
		Matrix< int > D = factory::dense< int >( n, 1, IOMode::SEQUENTIAL, 2 );
		std::vector< bool > touched( n, false );
		for( const auto & e : D ) {
			touched[ e.first.first ] = true;
			if( e.first.second != 0 ) {
				std::cerr << "Test FAILED: grb::factory::identity<int>[n,1]: incorrect column coordinate\n";
				rc = FAILED;
				return;
			}
			if( e.second != 2 ) {
				std::cerr << "Test FAILED: grb::factory::identity<int>[n,1]: incorrect value\n";
				rc = FAILED;
				return;
			}
		}
		if( std::find( touched.begin(), touched.end(), false ) != touched.end() ) {
			std::cerr << "Test FAILED: grb::factory::identity<int>[n,1]: not dense\n";
			rc = FAILED;
			return;
		}
	}

	rc = SUCCESS;
}

int main( int argc, char ** argv ) {
	// defaults
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is " << in << "): a positive integer.\n";
		return 1;
	}
	if( argc == 2 ) {
		in = atoi( argv[ 1 ] );
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
