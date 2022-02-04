
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
#include <string>
#include <type_traits>
#include <vector>

#include <graphblas.hpp>

template< typename StructuredMat >
void ask_questions( const StructuredMat & M, std::string name ) {

	using M_type = typename std::remove_const< typename std::remove_reference< decltype( M ) >::type >::type;

	std::cout << name << "( " << grb::nrows( M ) << ", " << grb::ncols( M ) << " )" << std::endl;
	std::cout << "Is " << name << ":" << std::endl;
	std::cout << "\ta structured Matrix? " << grb::is_structured_matrix< M_type >::value << std::endl;
	std::cout << "\tgeneral? " << grb::structures::is_a< M_type, grb::structures::General >::value << std::endl;
	std::cout << "\tsquare? " << grb::structures::is_a< M_type, grb::structures::Square >::value << std::endl;
	std::cout << "\tfull rank? " << grb::structures::is_a< M_type, grb::structures::FullRank >::value << std::endl;
	std::cout << "\tnon-singular? " << grb::structures::is_a< M_type, grb::structures::NonSingular >::value << std::endl;
}

void grb_program( const size_t & n, grb::RC & rc ) {

	std::cout << "\tStarting structured matrices test with size: " << n << "\n";

	// initialize test
	grb::StructuredMatrix< float, grb::structures::General > M( n, 2 * n );
	grb::StructuredMatrix< float, grb::structures::Square > A( n );
	grb::StructuredMatrix< float, grb::structures::NonSingular > B( n, n );
	grb::StructuredMatrix< float, grb::structures::FullRank > C( n, 2 * n );
	decltype( A )::transpose_t At( A );
	decltype( M )::transpose_t Mt( M );

	grb::get_ref< decltype( M ) >::type Mref( M );
	grb::get_ref< decltype( M ), grb::structures::Square >::type Sq_Mref( M );

	grb::remove_ref< decltype( Mt ) >::type M1( n, n );

	ask_questions( M, "M" );
	ask_questions( A, "A" );
	ask_questions( B, "B" );
	ask_questions( C, "C" );

	ask_questions( At, "At" );
	ask_questions( Mt, "Mt" );
	ask_questions( Mref, "Mref" );

	rc = grb::SUCCESS;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 5;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		size_t read;
		std::istringstream ss( argv[ 1 ] );
		if( ! ( ss >> read ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( ! ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( read % 2 != 0 ) {
			std::cerr << "Given value for n is odd\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer, the "
					 "test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< grb::AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != grb::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != grb::SUCCESS ) {
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
